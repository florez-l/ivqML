// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Linear__hxx__
#define __PUJ_ML__Model__Linear__hxx__

// -------------------------------------------------------------------------
namespace PUJ_ML::Model::
{
namespace Model
{
  /**
   */
  template< class _R >
  class Linear
    : public Model::Base< _R >
  {
  public:
    using Self = Linear;
    using Superclass = Model::Base< _R >;
    using TReal = typename Superclass::TReal;
    using TMatrix = typename Superclass::TMatrix;
    using TCol = typename Superclass::TCol;
    using TRow = typename Superclass::TRow;

  public:
    Linear( const unsigned long long& n = 1 )
      : Superclass( n )
      {
      }
    virtual ~Linear( )
      {
        if( this->m_T != nullptr )
          delete this->m_T;
      }

    unsigned long long number_of_parameters( ) const
      {
        return( this->m_P.size( ) );
      }

    virtual void set_number_of_parameters(
      const unsigned long long& n
      ) override
      {
        this->Superclass::set_number_of_parameters( n + 1 );
        this->m_T =
          new Eigen::Map< TCol >(
            this->m_P.data( ) + 1, this->m_P.size( ) - 1, 1
            );
      }

    template< class _Y, class _X >
    void evaluate(
      Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
      ) const
      {
        Y.derived( ) = ( ( X.derived( ).template cast< _R >( ) * *( this->m_T ) ).array( ) + this->m_P[ 0 ] ).template cast< typename _Y::Scalar >( );
      }

  protected:
    Eigen::Map< TCol >* m_T { nullptr };
  };
} // end namespace
} // end namespace

#endif // __PUJ_ML__Model__Linear__hxx__

// eof - $RCSfile$
