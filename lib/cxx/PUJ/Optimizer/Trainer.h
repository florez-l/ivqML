// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Optimizer__Trainer__h__
#define __PUJ__Optimizer__Trainer__h__

#include <boost/program_options.hpp>

namespace PUJ
{
  namespace Optimizer
  {
    /**
     */
    template< class _TOptimizer >
    class Trainer
    {
    public:
      using TOptimizer = _TOptimizer;
      using TModel     = typename TOptimizer::TModel;
      using TCost      = typename TModel::Cost;
      using TMatrix    = typename TModel::TMatrix;
      using TScalar    = typename TModel::TScalar;

    public:
      Trainer( )
      {
        this->_Init( );
      }

      template< class _TValue, class ... _TArgs >
      Trainer(
        const std::string& name, const _TValue& default_value,
        _TArgs ...args
        )
      {
        this->_Init( );
        this->_CompleteOptions( name, default_value, args... );
      }
      virtual ~Trainer( )
      {
        delete this->m_Options;
      }
      
      bool ParseArguments( int argc, char** argv )
      {
        boost::program_options::store( boost::program_options::parse_command_line( argc, argv, *this->m_Options ), this->m_VariablesMap );
        boost::program_options::notify( this->m_VariablesMap );

        if( this->m_VariablesMap.count( "help" ) )
        {
          std::cerr << *this->m_Options << "\n";
          return( false );
        } // end if
        return( true );
      }
      
      template< class _TValue >
      const _TValue& GetParameter( const std::string& name ) const
      {
        return( this->m_VariablesMap[ name ].as< _TValue >( ) );
      }
      
      void SetTrainData( const TMatrix& X, const TMatrix& Y )
      {
        this->m_Cost.SetTrainData( X, Y );
      }

      void SetDebug( typename TOptimizer::TDebug d )
      {
        this->m_Optimizer.SetDebug( d );
      }
      
      void Fit( )
      {
        this->m_Optimizer.Fit( );
      }
      
    protected:
      void _Init( )
      {
        this->m_Options =
          new boost::program_options::options_description( "Trainer options" );

        this->m_Options->add_options( )( "help", "produce help message" );

        this->m_Cost.SetModel( &this->m_Model );
        this->m_Optimizer.SetCost( &this->m_Cost );
        this->m_Cost.AddArguments( this->m_Options );
        this->m_Optimizer.AddArguments( this->m_Options );
      }
      
      void _CompleteOptions( )
      {
      }

      template< class _TValue, class ... _TArgs >
      void _CompleteOptions(
        const std::string& name, const _TValue& default_value,
        _TArgs ...args
        )
      {
        this->m_Options->add_options( )
         ( name.c_str( ),
           boost::program_options::value< _TValue >( )->
           default_value( default_value ),
           "???"
         );
        this->_CompleteOptions( args... );
      }
        
    protected:
      boost::program_options::options_description* m_Options;
      boost::program_options::variables_map m_VariablesMap;

      TModel     m_Model;
      TCost      m_Cost;
      TOptimizer m_Optimizer;
    };
  } // end namespace
} // end namespace

#endif // __PUJ__Optimizer__GradientDescent__h__

// eof - $RCSfile$
