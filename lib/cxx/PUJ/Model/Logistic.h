// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Logistic__h__
#define __PUJ__Logistic__h__

#include <vector>
#include <PUJ/Model/Linear.h>

namespace PUJ
{
  namespace Model
  {
    /**
     */
    template< class _TScalar, class _TTraits = PUJ::Traits< _TScalar > >
    class Logistic
    {
    public:
      PUJ_TraitsMacro( Logistic );
      using TLinear = PUJ::Model::Linear< _TScalar, _TTraits >;

    public:
      Logistic( const TRow& w, const TScalar& b );
      virtual ~Logistic( );

      unsigned long GetDimensions( ) const;
      const TRow& GetWeights( ) const;
      const TScalar& GetBias( ) const;

      void SetWeights( const TRow& w );
      void SetBias( const TScalar& b );

      TScalar operator()( const TRow& x, bool threshold = true ) const;
      TCol operator()( const TMatrix& x, bool threshold = true ) const;

    protected:
      void _StreamIn( std::istream& i );
      void _StreamOut( std::ostream& o ) const;

    protected:
      TLinear* m_Linear;

    public:
      /**
       */
      class Cost
      {
      public:
        Cost( const TMatrix& X, const TCol& y );
        virtual ~Cost( ) = default;

        TScalar operator()( const TRow& t, TRow* g = nullptr ) const;

      protected:
        TMatrix m_X;
        TRow    m_y;
        TRow    m_Xy;
        TScalar m_uy;
        std::vector< unsigned long long > m_Zeros;
        std::vector< unsigned long long > m_Ones;
      };

    public:
      friend std::istream& operator>>( std::istream& i, Self& m )
        {
          i >> *( m.m_Linear );
          return( i );
        }
      friend std::ostream& operator<<( std::ostream& o, const Self& m )
        {
          o << *( m.m_Linear );
          return( o );
        }
    };
  } // end namespace
} // end namespace

#endif // __PUJ__Logistic__h__

// eof - $RCSfile$
